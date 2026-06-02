(define (problem picture_12)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery green_sensor red_regulator - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear blue_battery)
        (clear red_regulator)
        (clear green_sensor)
        (part_at red_regulator table)
        (part_at green_sensor table)
        (part_at blue_battery table)
    )
    
    (:goal
        (and
            (part_at red_regulator regulator_placement)
            (part_at blue_battery battery_placement)
        )
    )
)
