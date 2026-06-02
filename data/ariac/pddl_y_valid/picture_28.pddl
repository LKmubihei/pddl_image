(define (problem picture_28)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_pump red_battery green_regulator blue_battery - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear red_pump)
        (clear red_battery)
        (clear blue_battery)
        (part_at red_pump table)
        (part_at green_regulator table)
        (part_at red_battery table)
        (on blue_battery green_regulator )
    )
    
    (:goal
        (and
            (part_at red_pump pump_placement)
            (part_at green_regulator regulator_placement)
            (part_at red_battery battery_placement)
        )
    )
)
