(define (problem picture_190)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery green_battery  red_pump red_pump_1 green_regulator blue_regulator - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear green_battery)
        (clear blue_battery)
        (clear green_regulator)
        (clear blue_regulator)
        (clear red_pump)
        (on blue_battery red_pump_1)
        (part_at red_pump_1 table)
        (part_at green_battery table)
        (part_at green_regulator table)
        (part_at blue_regulator table)
        (part_at red_pump pump_placement)  
    )
    
    (:goal
        (and
            (part_at blue_battery battery_placement)
            (part_at green_regulator regulator_placement)
        )
    )
)